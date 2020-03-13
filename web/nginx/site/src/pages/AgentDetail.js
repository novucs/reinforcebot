import React, {Component} from "react";
import TopMenu from "../TopMenu";
import {Breadcrumb, Button, Container, Divider, Grid, Header, List, Loader, Segment, Sticky} from "semantic-ui-react";
import Footer from "../Footer";
import logo from "../icon.svg";
import {BASE_URL, ensureSignedIn, fetchUsers, getJWT, hasJWT, refreshJWT} from "../Util";
import Moment from 'moment';

export default class AgentDetail extends Component {
  constructor(props) {
    super(props);
    this.state = {
      agent: null,
      users: {},
    };
  }

  componentDidMount() {
    ensureSignedIn();
    this.fetchAgent();
  }

  fetchAgent() {
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
  }

  agentHistory() {
    let history = [];
    this.state.agent.history.forEach(item => {
      if (!(item.history_user_id in this.state.users)) {
        return;
      }
      let user = this.state.users[item.history_user_id];
      Moment.locale('en');
      history.push((
        <List.Item key={'history-' + item.id}>
          <a download href={BASE_URL + '/api/media/' + item.parameters}>
            {Moment(item.history_date).format('LLLL') + ' authored by ' + user.username}
          </a>
        </List.Item>
      ));
    });
    return history;
  }

  agentContent() {
    return (
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
              {this.state.agent.description}
            </Segment>
            <Divider>History</Divider>
            <div style={{textAlign: 'left'}}>
              <List>
                {this.agentHistory()}
              </List>
            </div>
          </Grid.Column>
          <Grid.Column className='five wide'>
            <Sticky>
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
                <Button
                  fluid
                  style={{marginTop: '5px'}}
                  icon='tag'
                  content='Edit Name'
                />
                <Button
                  fluid
                  style={{marginTop: '5px'}}
                  icon='pencil'
                  content='Edit Description'
                />
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
            </Sticky>
          </Grid.Column>
        </Grid>
      </div>
    )
  }

  render() {
    return (
      <div className="SitePage">
        <TopMenu/>
        <Container className="SiteContents" style={{marginTop: '80px'}}>
          {this.state.agent !== null ? this.agentContent() : (
            <Loader/>
          )}
        </Container>
        < Footer/>
      </div>
    )
  }
}
