import React, {Component} from "react";
import TopMenu from "../TopMenu";
import {
  Breadcrumb,
  Button,
  Container,
  Divider,
  Form,
  Grid,
  Header,
  Icon,
  List,
  Loader,
  Modal,
  Segment
} from "semantic-ui-react";
import Footer from "../Footer";
import logo from "../icon.svg";
import {BASE_URL, ensureSignedIn, fetchUsers, getJWT, hasJWT, refreshJWT} from "../Util";
import Moment from 'moment';
import {SemanticToastContainer, toast} from "react-semantic-toasts";

export default class AgentDetail extends Component {
  constructor(props) {
    super(props);
    this.state = {
      agent: null,
      users: {},
      editingName: false,
      editingDescription: false,
      updatingModal: false,
      deleting: false,
    };
  }

  componentDidMount = () => {
    ensureSignedIn();
    this.fetchAgent();
  };

  fetchAgent = () => {
    if (!hasJWT()) {
      return;
    }

    fetch(BASE_URL + '/api/agents/' + this.props.match.params.id + '/', {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': 'JWT ' + getJWT(),
      },
    }).then(response => {
      if (response.status === 401) {
        refreshJWT();
        return;
      }

      if (response.status !== 200) {
        response.text().then(body => {
          console.error("Unable to fetch agent: ", body);
        });
        return;
      }

      response.json().then(agent => {
        this.setState({agent: agent});

        let userURIs = [agent.author];
        agent.history.forEach(h => {
          userURIs.push(BASE_URL + '/api/auth/users/' + h.history_user_id + '/');
        });

        fetchUsers(userURIs, users => {
          this.setState({users: users});
        });
      });
    });
  };

  agentHistory = () => {
    let history = [];
    this.state.agent.history.forEach(item => {
      if (!(item.history_user_id in this.state.users)) {
        return;
      }
      let user = this.state.users[item.history_user_id];
      Moment.locale('en');
      history.push((
        <List.Item key={'history-' + item.history_id}>
          <a download href={BASE_URL + '/api/media/' + item.parameters}>
            {Moment(item.history_date).format('LLLL') + ' authored by ' + user.username}
          </a>
        </List.Item>
      ));
    });
    return history;
  };

  closeEditWindow = () => {
    this.setState({
      editingName: false,
      editingDescription: false,
      updatingModal: false,
      deleting: false,
    });
  };

  editAgent = fieldName => {
    if (!hasJWT() || this.state.agent[fieldName] === '') {
      return;
    }

    let body = {};
    body[fieldName] = this.state.agent[fieldName];

    fetch(BASE_URL + '/api/agents/' + this.state.agent.id + '/', {
      method: 'PATCH',
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': 'JWT ' + getJWT(),
      },
      body: JSON.stringify(body)
    }).then(response => {
      if (response.status === 401) {
        refreshJWT();
        return;
      }

      if (response.status !== 200) {
        response.text().then(body => {
          console.error("Unable to update agent: ", body);
        });
        return;
      }

      toast(
        {
          type: 'success',
          title: 'Success',
          description: <p>Agent successfully updated</p>
        },
      );
      this.closeEditWindow();
      this.fetchAgent();
    });
  };

  editNameModal = () => (
    <Modal open={this.state.editingName}
           trigger={
             <Button
               fluid
               style={{marginTop: '5px'}}
               icon='tag'
               content='Edit Name'
               onKeyDown={() => this.editAgent('name')}
               onClick={() => this.setState({editingName: true})}
             />
           }
           basic
           size='small'>
      <Header icon='tag' content='Editing Agent Name'/>
      <Modal.Content>
        <Form.Input
          fluid
          required
          icon="tag"
          iconPosition="left"
          placeholder="Name"
          defaultValue={this.state.agent.name}
          onChange={event => this.setState({
            agent: {...this.state.agent, name: event.target.value}
          })}
        />
      </Modal.Content>
      <Modal.Actions>
        <Button basic color='red' inverted onClick={() => this.closeEditWindow()}>
          <Icon name='remove'/> Cancel
        </Button>
        <Button
          color='green'
          inverted
          disabled={this.state.agent.name === ''}
          onClick={() => this.editAgent('name')}
        >
          <Icon name='checkmark'/> Submit
        </Button>
      </Modal.Actions>
    </Modal>
  );

  editDescriptionModal = () => (
    <Modal open={this.state.editingDescription}
           trigger={
             <Button
               fluid
               style={{marginTop: '5px'}}
               icon='pencil'
               content='Edit Description'
               onKeyDown={() => this.editAgent('description')}
               onClick={() => this.setState({editingDescription: true})}
             />
           }
           basic
           size='small'>
      <Header icon='pencil' content='Editing Agent Description'/>
      <Modal.Content>
        <Form.TextArea
          style={{width: '100%'}}
          rows={10}
          required
          placeholder="Description"
          defaultValue={this.state.agent.description}
          onChange={event => this.setState({
            agent: {...this.state.agent, description: event.target.value}
          })}
        />
      </Modal.Content>
      <Modal.Actions>
        <Button basic color='red' inverted onClick={() => this.closeEditWindow()}>
          <Icon name='remove'/> Cancel
        </Button>
        <Button
          color='green'
          inverted
          disabled={this.state.agent.description === ''}
          onClick={() => this.editAgent('description')}
        >
          <Icon name='checkmark'/> Submit
        </Button>
      </Modal.Actions>
    </Modal>
  );

  descriptionLines = () => {
    let lines = [];
    let i = 0;
    this.state.agent.description.split('\n').forEach(line => {
      i += 1;
      lines.push((
        <List.Item key={'line-' + i}>{line}</List.Item>
      ))
    });
    return lines;
  };

  agentContent = () => (
    <div>
      <Header as="h2" color="teal" textAlign="center">
        <img src={logo} alt="logo" className="image"/>{" "}
        {this.state.agent.name}
      </Header>
      <Grid>
        <Grid.Column className='eleven wide'>
          <Segment textAlign='left'>
            <Breadcrumb icon='right angle' sections={[
              {key: 'Dashboard', content: 'Dashboard', href: '/dashboard'},
              {key: 'Agent', content: 'Agent', active: true},
            ]}/>
            <Divider/>
            <List>
              {this.descriptionLines()}
            </List>
          </Segment>
          <Divider>History</Divider>
          <div style={{textAlign: 'left'}}>
            <List>
              {this.agentHistory()}
            </List>
          </div>
        </Grid.Column>
        <Grid.Column className='five wide'>
          <Segment>
            <Header as='h4'>Options</Header>
            <Divider/>
            <Button
              fluid
              primary
              download
              href={this.state.agent.parameters}
              icon='cloud download'
              content='Download'
            />
            {this.editNameModal()}
            {this.editDescriptionModal()}
            <Button
              fluid
              style={{marginTop: '5px'}}
              color='yellow'
              icon='cog'
              content='Update Model'
            />
            <Button
              fluid
              style={{marginTop: '5px'}}
              color='red'
              icon='cancel'
              content='Delete'
            />
          </Segment>
        </Grid.Column>
      </Grid>
    </div>
  );

  render = () => (
    <div className="SitePage">
      <TopMenu/>
      <SemanticToastContainer position='bottom-right'/>
      <Container className="SiteContents" style={{marginTop: '80px'}}>
        {this.state.agent !== null ? this.agentContent() : (
          <Loader/>
        )}
      </Container>
      < Footer/>
    </div>
  );
}
