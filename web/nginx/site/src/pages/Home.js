import React, {Component} from 'react'
import {Button, Container, Grid, Header, Icon, Image, Segment} from 'semantic-ui-react'
import whiteLogo from "../white-logo.svg";
import problemImg from "../problem.png";
import solutionImg from "../solution.png";
import Footer from "../components/Footer";
import TopMenu from "../components/TopMenu";
import {fetchMe} from "../Util";


export default class HomepageLayout extends Component {
  constructor(props) {
    super(props);
    this.state = {};
  }

  componentDidMount() {
    fetchMe(me => this.setState({me}));
  }

  render = () => (
    <div className="SitePage">
      <TopMenu me={this.state.me}/>
      <div className="SiteContents">
        <Segment
          inverted
          textAlign='center'
          style={{minHeight: 700, padding: '1em 0em'}}
          vertical
        >
          <Container text>
            <img
              style={{
                marginBottom: '3em',
                marginTop: '12em'
              }}
              src={whiteLogo}
              alt="logo"
              className="image"
            />
            <Button primary href='/start' size='huge'>
              Get Started
              <Icon name='right arrow'/>
            </Button>
          </Container>
        </Segment>

        <Segment style={{padding: '8em 0em'}} vertical>
          <Grid container stackable verticalAlign='middle'>
            <Grid.Row>
              <Grid.Column width={8}>
                <Header as='h3' style={{fontSize: '2em'}}>
                  Ever been tired of repetitive tasks?
                </Header>
                <p style={{fontSize: '1.33em'}}>
                  Tasks on your computer may at time to time become repetitive.

                  The only two solutions around include creating macros, and developing an agent to automate the
                  workflow.

                  However, macros are brittle, and hiring a developer to create an agent can be expensive.
                </p>
              </Grid.Column>
              <Grid.Column floated='right' width={6}>
                <Image size='medium' src={problemImg}/>
              </Grid.Column>
            </Grid.Row>
          </Grid>
        </Segment>

        <Segment style={{padding: '8em 0em'}} vertical>
          <Grid container stackable verticalAlign='middle'>
            <Grid.Row>
              <Grid.Column floated='left' width={6}>
                <Image size='medium' src={solutionImg}/>
              </Grid.Column>
              <Grid.Column width={8}>
                <Header as='h3' style={{fontSize: '2em'}}>
                  ReinforceBot automates your tasks
                </Header>
                <p style={{fontSize: '1.33em'}}>
                  ReinforceBot enables you to create custom agents with no prior
                  development experience required. These agents will be happy to
                  automate your task for you, and are even capable of playing games!
                </p>
              </Grid.Column>
            </Grid.Row>
          </Grid>
        </Segment>
      </div>
      <Footer/>
    </div>
  );
}
